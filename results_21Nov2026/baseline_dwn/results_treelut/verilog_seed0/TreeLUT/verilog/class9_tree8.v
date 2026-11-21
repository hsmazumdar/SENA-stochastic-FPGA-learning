module class9_tree8(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[423] & (~i[27])) | (i[408] & i[200] & (~i[326]) & i[27]) | ((~i[256]) & (~i[200]) & (~i[326]) & i[27]) | ((~i[272]) & (~i[229]) & i[394] & i[326] & i[27]) | (i[173] & i[210] & (~i[394]) & i[326] & i[27]) | ((~i[173]) & i[210] & (~i[394]) & i[326] & i[27]) | (i[216] & (~i[210]) & (~i[394]) & i[326] & i[27]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
