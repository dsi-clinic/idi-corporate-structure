import pandas as pd

if __name__ == "__main__":
    cik_to_lei = pd.read_csv(
        "~/Box/dsi-core/11th-hour/idi-corporate-structure/cik-to-lei.csv", dtype=str
    )

    entities = pd.read_csv(
        "~/Box/dsi-core/11th-hour/idi-corporate-structure/gleif/20250807-0800-gleif-goldencopy-lei2-golden-copy.csv",
        usecols=["LEI", "Entity.LegalName"],
        dtype=str,
    ).set_index("LEI")

    top_level = []
    for row in cik_to_lei.itertuples():
        if isinstance(row.lei, str):
            top_level.append(
                (
                    row.cik,
                    row.lei,
                    f"{entities.loc[row.lei].item()}{' (PUBLIC)' if row.is_public == 'True' else ' '} https://permid.org/{row.permid}",
                )
            )
    top_level.sort(key=lambda x: x[2])

    relations = (
        pd.read_csv(
            "~/Box/dsi-core/11th-hour/idi-corporate-structure/gleif/20250807-0800-gleif-goldencopy-rr-golden-copy.csv",
            usecols=[
                "Relationship.StartNode.NodeID",
                "Relationship.EndNode.NodeID",
                "Relationship.RelationshipType",
                "Relationship.RelationshipStatus",
            ],
            dtype=str,
        )
        .rename(
            columns={
                "Relationship.StartNode.NodeID": "start",
                "Relationship.EndNode.NodeID": "end",
                "Relationship.RelationshipType": "type",
                "Relationship.RelationshipStatus": "status",
            }
        )
        .query("status == 'ACTIVE'")
    )

    consolidates = {}
    fund_manages = {}
    has_subfund = {}
    has_intbranch = {}
    has_feeder = {}
    for row in relations.itertuples():
        if row.end not in consolidates:
            consolidates[row.end] = []
            fund_manages[row.end] = []
            has_subfund[row.end] = []
            has_intbranch[row.end] = []
            has_feeder[row.end] = []
        # row.end is the parent: "row.start IS CONSOLIDATED BY row.end"
        if row.type == "IS_DIRECTLY_CONSOLIDATED_BY":
            consolidates[row.end].append(row.start)
        elif row.type == "IS_FUND-MANAGED_BY":
            fund_manages[row.end].append(row.start)
        elif row.type == "IS_HAS_SUBFUND_OF":
            has_subfund[row.end].append(row.start)
        elif row.type == "IS_INTERNATIONAL_BRANCH_OF":
            has_intbranch[row.end].append(row.start)
        elif row.type == "IS_FEEDER_TO":
            has_feeder[row.end].append(row.start)

    def children_of(lei):
        return (
            [
                (x, f"CONSOLIDATES: {entities.loc[x].item()}")
                for x in consolidates.get(lei, [])
            ]
            + [
                (x, f"FUND-MANAGES: {entities.loc[x].item()}")
                for x in fund_manages.get(lei, [])
            ]
            + [
                (x, f"HAS SUBFUND: {entities.loc[x].item()}")
                for x in has_subfund.get(lei, [])
            ]
            + [
                (x, f"HAS INTERNATIONAL BRANCH: {entities.loc[x].item()}")
                for x in has_intbranch.get(lei, [])
            ]
            + [
                (x, f"HAS FEEDER: {entities.loc[x].item()}")
                for x in has_feeder.get(lei, [])
            ]
        )

    ELBOW = "└──"
    PIPE = "│  "
    TEE = "├──"
    BLANK = "   "

    def draw_tree(lei, line, last, header, seen):
        print(
            f"{header}{ELBOW if last else TEE}{line}{' (SAME AS ABOVE)' if lei in seen else ''}"
        )
        if lei not in seen:
            seen.add(lei)
            children = children_of(lei)
            children.sort(key=lambda x: x[1])
            for i, (sublei, subline) in enumerate(children):
                draw_tree(
                    sublei,
                    subline,
                    i == len(children) - 1,
                    header + (BLANK if last else PIPE),
                    seen,
                )

    for i, (cik, lei, line) in enumerate(top_level):
        seen = set()
        print(f"CIK{cik}: {line}")
        children = children_of(lei)
        children.sort(key=lambda x: x[1])
        for i, (sublei, subline) in enumerate(children):
            draw_tree(sublei, subline, i == len(children) - 1, "", seen)
        print()
